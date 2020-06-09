//
//  _ConflatedChannel.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

fileprivate struct Tag: OptionSet {
    let rawValue: Int
    
    static let hasWaiting = Tag(rawValue: 1 << 0)
    static let closed = Tag(rawValue: 1 << 1)
    static let canceled = Tag(rawValue: 1 << 2)
}

internal final class _ConflatedChannel<T>: _Channel<T> {
    
    private typealias Tagged = TaggedPointer<Tag, T>
    private typealias ReceiveCallback = (Result<T, CoChannelError>) -> Void
    
    private var receiveCallbacks = FifoQueue<ReceiveCallback>()
    private var rawValue = 0
    
    internal override var bufferType: CoChannel<T>.BufferType {
        .conflated
    }
    
    private func cas(oldValue: Int, tagged: Tagged) -> Bool {
        atomicCAS(&rawValue, expected: oldValue, desired: tagged.rawValue)
    }
    
    // MARK: - send
    
    internal override func awaitSend(_ element: T) throws {
        var pointer: UnsafeMutablePointer<T>?
        while true {
            let rawValue = self.rawValue
            var tagged = Tagged(rawValue: rawValue)
            if tagged[.canceled] {
                pointer?.deinitialize(count: 1).deallocate()
                throw CoChannelError.canceled
            } else if tagged[.closed] {
                pointer?.deinitialize(count: 1).deallocate()
                throw CoChannelError.closed
            } else if tagged[.hasWaiting] {
                tagged.counter -= 1
                if tagged.counter == 0 { tagged[.hasWaiting] = false }
                if !cas(oldValue: rawValue, tagged: tagged) { continue }
                pointer?.deinitialize(count: 1).deallocate()
                return receiveCallbacks.blockingPop()(.success(element))
            } else if pointer == nil {
                pointer = .allocate(capacity: 1)
                pointer?.initialize(to: element)
            }
            let address = tagged.pointerAddress
            tagged.pointer = pointer
            if !cas(oldValue: rawValue, tagged: tagged) { continue }
            UnsafeMutablePointer<T>(bitPattern: address)?.deinitialize(count: 1).deallocate()
            return
        }
    }
    
    internal override func sendFuture(_ future: CoFuture<T>) {
        future.whenSuccess { [weak self] in try? self?.awaitSend($0) }
    }
    
    internal override func offer(_ element: T) -> Bool {
        (try? awaitSend(element)) != nil
    }
    
    // MARK: - receive
    
    internal override func whenReceive(_ callback: @escaping (Result<T, CoChannelError>) -> Void) {
        while true {
            let rawValue = self.rawValue
            var tagged = Tagged(rawValue: rawValue)
            if tagged[.canceled] { return callback(.failure(.canceled)) }
            let address = tagged.pointerAddress
            if tagged[.closed] {
                if address == 0 { return callback(.failure(.closed)) }
            } else if tagged[.hasWaiting] || address == 0 {
                tagged.counter += 1
                tagged[.hasWaiting] = true
                if !cas(oldValue: rawValue, tagged: tagged) { continue }
                return receiveCallbacks.push(callback)
            }
            tagged.pointer = nil
            if !cas(oldValue: rawValue, tagged: tagged) { continue }
            let pointer = UnsafeMutablePointer<T>(bitPattern: address)!
            defer { pointer.deinitialize(count: 1).deallocate() }
            defer { if tagged[.closed] { finish() } }
            return callback(.success(pointer.pointee))
        }
    }
    
    internal override func poll() -> T? {
        while true {
            let rawValue = self.rawValue
            var tagged = Tagged(rawValue: rawValue)
            if tagged[[.canceled, .hasWaiting]] { return nil }
            let address = tagged.pointerAddress
            if address == 0 { return nil }
            tagged.pointer = nil
            if !cas(oldValue: rawValue, tagged: tagged) { continue }
            let pointer = UnsafeMutablePointer<T>(bitPattern: address)
            defer { pointer?.deinitialize(count: 1).deallocate() }
            defer { if tagged[.closed] { finish() } }
            return pointer?.pointee
        }
    }
    
    internal override func awaitReceive() throws -> T {
        while true {
            let rawValue = self.rawValue
            var tagged = Tagged(rawValue: rawValue)
            if tagged[.canceled] { throw CoChannelError.canceled }
            let address = tagged.pointerAddress
            if tagged[.closed] {
                if address == 0 { throw CoChannelError.closed }
            } else if tagged[.hasWaiting] || address == 0 {
                tagged.counter += 1
                tagged[.hasWaiting] = true
                if !cas(oldValue: rawValue, tagged: tagged) { continue }
                return try Coroutine.await { receiveCallbacks.push($0) }.get()
            }
            tagged.pointer = nil
            if !cas(oldValue: rawValue, tagged: tagged) { continue }
            let pointer = UnsafeMutablePointer<T>(bitPattern: address)!
            defer { pointer.deinitialize(count: 1).deallocate() }
            defer { if tagged[.closed] { finish() } }
            return pointer.pointee
        }
    }
    
    internal override var count: Int {
        isEmpty ? 0 : 1
    }
    
    internal override var isEmpty: Bool {
        let tagged = Tagged(rawValue: rawValue)
        return tagged[[.canceled, .hasWaiting]] || tagged.pointerAddress == 0
    }
    
    // MARK: - close
    
    internal override func close() -> Bool {
        while true {
            let rawValue = self.rawValue
            var taggedPointer = Tagged(rawValue: rawValue)
            if taggedPointer[[.canceled, .closed]] { return false }
            taggedPointer[.closed] = true
            if taggedPointer[.hasWaiting] {
                let count = taggedPointer.counter
                taggedPointer[.hasWaiting] = false
                taggedPointer.counter = 0
                if !cas(oldValue: rawValue, tagged: taggedPointer) { continue }
                for _ in 0..<count { receiveCallbacks.blockingPop()(.failure(.closed)) }
                finish()
                return true
            } else if cas(oldValue: rawValue, tagged: taggedPointer) {
                if taggedPointer.pointerAddress == 0 { finish() }
                return true
            }
        }
    }
    
    internal override var isClosed: Bool {
        Tagged(rawValue: rawValue)[.closed]
    }
    
    // MARK: - cancel
    
    internal override func cancel() {
        while true {
            let tagged = Tagged(rawValue: rawValue)
            if tagged[.canceled] { return }
            let desired = Tag.canceled.rawValue
            if tagged[.hasWaiting] {
                let count = tagged.counter
                if !atomicCAS(&rawValue, expected: tagged.rawValue, desired: desired) { continue }
                for _ in 0..<count { receiveCallbacks.blockingPop()(.failure(.canceled)) }
                return finish()
            }
            let address = tagged.pointerAddress
            if !atomicCAS(&rawValue, expected: tagged.rawValue, desired: desired) { continue }
            UnsafeMutablePointer<T>(bitPattern: address)?.deinitialize(count: 1).deallocate()
            return finish()
        }
    }
    
    internal override var isCanceled: Bool {
        Tagged(rawValue: rawValue)[.canceled]
    }
    
    deinit {
        let tagged = Tagged(rawValue: rawValue)
        if tagged[.hasWaiting] {
            while let block = receiveCallbacks.pop() {
                block(.failure(.canceled))
            }
        } else if let pointer = tagged.pointer {
            pointer.deinitialize(count: 1).deallocate()
        }
        receiveCallbacks.free()
    }
    
}
