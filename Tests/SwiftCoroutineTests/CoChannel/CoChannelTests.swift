//
//  CoChannelTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 20.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoChannelTests: XCTestCase {
    
    func testSequence() {
        let exp = expectation(description: "testSequence")
        let (receiver, sender) = CoChannel<Int>().pair
        var set = Set<Int>()
        DispatchQueue.global().startCoroutine {
            for value in receiver.makeIterator() {
                set.insert(value)
            }
            XCTAssertEqual(set.count, 100_000)
            exp.fulfill()
        }
        DispatchQueue.concurrentPerform(iterations: 100_000) { index in
            sender.offer(index)
        }
        sender.close()
        wait(for: [exp], timeout: 10)
    }
    
    func testMeasure() {
        let group = DispatchGroup()
        measure {
            group.enter()
            let (receiver, sender) = CoChannel<Int>(maxBufferSize: 1).pair
            DispatchQueue.global().startCoroutine {
                for value in receiver.makeIterator() {
                    let _ = value
                }
                group.leave()
            }
            DispatchQueue.global().startCoroutine {
                for index in (0..<100_000) {
                    try sender.awaitSend(index)
                }
                sender.close()
            }
            group.wait()
        }
    }
    
    func testInsideCoroutine() {
        let exp = expectation(description: "testInsideCoroutine")
        let (receiver, sender) = CoChannel<Int>(maxBufferSize: 1).pair
        var set = Set<Int>()
        DispatchQueue.global().startCoroutine {
            for value in receiver.makeIterator() {
                set.insert(value)
            }
            XCTAssertNil(try? receiver.awaitReceive())
            XCTAssertEqual(set.count, 100_000)
            exp.fulfill()
        }
        DispatchQueue.global().startCoroutine {
            for index in (0..<100_000) {
                try sender.awaitSend(index)
            }
            sender.close()
            try sender.awaitSend(-1)
        }
        wait(for: [exp], timeout: 20)
    }
    
    func testOffer() {
        let (receiver, sender) = CoChannel<Int>(maxBufferSize: 1).pair
        ImmediateScheduler().startCoroutine {
            XCTAssertEqual(try? receiver.awaitReceive(), 1)
        }
        sender.offer(1)
        sender.offer(2)
        sender.offer(3)
        XCTAssertEqual(sender.count, 1)
        receiver.whenReceive {
            XCTAssertEqual(try? $0.get(), 2)
        }
        sender.offer(4)
        XCTAssertEqual(receiver.poll(), 4)
        XCTAssertTrue(receiver.isEmpty)
        sender.offer(5)
        XCTAssertFalse(sender.isEmpty)
        XCTAssertEqual(receiver.makeIterator().next(), 5)
        XCTAssertNil(receiver.makeIterator().next())
        receiver.whenReceive {
            XCTAssertEqual(try? $0.get(), 6)
        }
        sender.offer(6)
    }
    
    func testFuture() {
        let (receiver, sender) = CoChannel<Int>(maxBufferSize: 1).pair
        receiver.receiveFuture().whenSuccess {
            XCTAssertEqual($0, 1)
        }
        sender.sendFuture(.init(result: .success(1)))
        sender.sendFuture(.init(result: .success(2)))
        sender.sendFuture(.init(result: .success(3)))
        receiver.receiveFuture().whenSuccess {
            XCTAssertEqual($0, 2)
        }
    }
    
    func testCancel() {
        let (receiver, sender) = CoChannel<Int>(maxBufferSize: 0).pair
        ImmediateScheduler().startCoroutine {
            XCTAssertThrowError(CoChannelError.canceled) { try sender.awaitSend(0) }
        }
        XCTAssertFalse(receiver.isCanceled)
        receiver.cancel()
        XCTAssertThrowError(CoChannelError.canceled) { try receiver.awaitReceive() }
        XCTAssertThrowError(CoChannelError.canceled) { try sender.awaitSend(1) }
        XCTAssertFalse(sender.offer(1))
        receiver.whenReceive { result in
            XCTAssertThrowError(CoChannelError.canceled) { try result.get() }
        }
        XCTAssertNil(receiver.poll())
        XCTAssertTrue(sender.isCanceled)
    }

    func testCancel2() {
        let exp = expectation(description: "testCancel2")
        let (receiver, sender) = CoChannel<Int>(maxBufferSize: 0).pair
        ImmediateScheduler().startCoroutine {
            XCTAssertThrowError(CoChannelError.canceled) { try receiver.awaitReceive() }
        }
        sender.cancel()
        sender.whenCanceled { exp.fulfill() }
        wait(for: [exp], timeout: 2)
    }
    
    func testClose() {
        let (receiver, sender) = CoChannel<Int>(maxBufferSize: 1).pair
        receiver.whenReceive { result in
            XCTAssertThrowError(CoChannelError.closed) { try result.get() }
        }
        XCTAssertFalse(receiver.isClosed)
        XCTAssertTrue(sender.close())
        XCTAssertFalse(sender.close())
        XCTAssertTrue(sender.isClosed)
        receiver.whenReceive { result in
            XCTAssertThrowError(CoChannelError.closed) { try result.get() }
        }
        receiver.receiveFuture().whenFailure {
            if let error = $0 as? CoChannelError {
                XCTAssertEqual(error, CoChannelError.closed)
            } else {
                XCTFail()
            }
        }
        sender.sendFuture(.init(result: .success(1)))
    }
    
    func testDeinit() {
        let exp = expectation(description: "testDeinit")
        var channel: CoChannel<Int>! = .init()
        weak var _weak = channel
        channel.offer(1)
        let promise = CoPromise<Int>()
        channel.sendFuture(promise)
        channel = nil
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
            promise.success(1)
            XCTAssertNil(_weak)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 3)
    }
    
    func testMap() {
        let exp = expectation(description: "testMap")
        let channel = CoChannel<Int>()
        let map2 = channel.map { $0 + 2 }
        channel.offer(7)
        XCTAssertEqual(map2.poll(), 9)
        let (receiver, sender) = channel.pair
        let map = receiver.map { $0 + 1 }.map { $0 + 1 }
        XCTAssertEqual(map.maxBufferSize, sender.maxBufferSize)
        sender.offer(1)
        sender.offer(2)
        XCTAssertFalse(map.isEmpty)
        XCTAssertFalse(map.isClosed)
        XCTAssertFalse(map.isCanceled)
        XCTAssertEqual(map.count, 2)
        XCTAssertEqual(map.poll(), 3)
        XCTAssertEqual(try? map.awaitReceive(), 4)
        sender.offer(3)
        sender.offer(4)
        XCTAssertEqual(map.receiveFuture().result, 5)
        map.whenReceive { XCTAssertEqual(try? $0.get(), 6) }
        XCTAssertNil(map.poll())
        sender.offer(5)
        map.makeIterator().forEach { XCTAssertEqual($0, 7) }
        sender.offer(6)
        Coroutine.start {
            map.makeIterator().forEach { XCTAssertEqual($0, 8) }
        }
        map.whenCanceled { exp.fulfill() }
        map.cancel()
        XCTAssertTrue(map.isCanceled)
        wait(for: [exp], timeout: 5)
    }
    
    func testCoChannelReceiverWrapper() {
        let wrapper = CoChannelReceiverWrapper<Any>()
        _ = try? wrapper.awaitReceive()
        _ = wrapper.receiveFuture()
        _ = wrapper.poll()
        wrapper.whenReceive { _ in }
        wrapper.cancel()
        wrapper.whenCanceled {}
        _ = wrapper.count
        _ = wrapper.isEmpty
        _ = wrapper.isClosed
        _ = wrapper.isCanceled
        _ = wrapper.maxBufferSize
    }
    
}
