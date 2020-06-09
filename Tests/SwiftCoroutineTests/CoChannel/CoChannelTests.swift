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
        let channel = CoChannel<Int>()
        var set = Set<Int>()
        DispatchQueue.global().startCoroutine {
            for value in channel.makeIterator() {
                set.insert(value)
            }
            XCTAssertEqual(set.count, 100_000)
            exp.fulfill()
        }
        DispatchQueue.concurrentPerform(iterations: 100_000) { index in
            channel.offer(index)
        }
        channel.close()
        wait(for: [exp], timeout: 10)
    }
    
    func testMeasure() {
        let group = DispatchGroup()
        measure {
            group.enter()
            let channel = CoChannel<Int>(capacity: 1)
            DispatchQueue.global().startCoroutine {
                for value in channel.makeIterator() {
                    let _ = value
                }
                group.leave()
            }
            DispatchQueue.global().startCoroutine {
                for index in (0..<100_000) {
                    try channel.awaitSend(index)
                }
                channel.close()
            }
            group.wait()
        }
    }
    
    func testBufferType() {
        XCTAssertEqual(CoChannel<Int>(capacity: 3).bufferType, .buffered(capacity: 3))
        XCTAssertEqual(CoChannel<Int>(bufferType: .buffered(capacity: 2)).bufferType, .buffered(capacity: 2))
        XCTAssertEqual(CoChannel<Int>(bufferType: .none).bufferType, .none)
        XCTAssertEqual(CoChannel<Int>(bufferType: .unlimited).bufferType, .unlimited)
        XCTAssertEqual(CoChannel<Int>(bufferType: .conflated).bufferType, .conflated)
        
        XCTAssertEqual(CoChannel<Int>(bufferType: .buffered(capacity: 2)).map { $0 }.bufferType, .buffered(capacity: 2))
        XCTAssertEqual(CoChannel<Int>(bufferType: .none).map { $0 }.bufferType, .none)
        XCTAssertEqual(CoChannel<Int>(bufferType: .unlimited).map { $0 }.bufferType, .unlimited)
        XCTAssertEqual(CoChannel<Int>(bufferType: .conflated).map { $0 }.bufferType, .conflated)
    }
    
    func testCoChannel() {
        let exp = expectation(description: "testCoChannel")
        exp.expectedFulfillmentCount = 6
        let channel = CoChannel<Int>()
        XCTAssertTrue(channel.offer(1))
        XCTAssertEqual(channel.count, 1)
        XCTAssertFalse(channel.isEmpty)
        XCTAssertEqual(channel.poll(), 1)
        channel.receiveFuture().whenSuccess {
            XCTAssertEqual($0, 2)
            exp.fulfill()
        }
        channel.sendFuture(.init(result: .success(2)))
        XCTAssertNotNil(try? channel.awaitSend(3))
        XCTAssertEqual(try? channel.awaitReceive(), 3)
        channel.whenReceive {
            XCTAssertEqual(try? $0.get(), 4)
            exp.fulfill()
        }
        XCTAssertTrue(channel.offer(4))
        channel.map { $0 + 1 }.whenReceive {
            XCTAssertEqual(try? $0.get(), 6)
            exp.fulfill()
        }
        XCTAssertTrue(channel.offer(5))
        channel.receiver.whenReceive {
            XCTAssertEqual(try? $0.get(), 6)
            exp.fulfill()
        }
        XCTAssertTrue(channel.sender.offer(6))
        XCTAssertTrue(channel.close())
        XCTAssertTrue(channel.isClosed)
        channel.cancel()
        XCTAssertTrue(channel.isCanceled)
        channel.whenCanceled { exp.fulfill() }
        channel.whenComplete { exp.fulfill() }
        wait(for: [exp], timeout: 1)
    }
    
    func testReceiveFuture() {
        let exp = expectation(description: "testReceiveFuture")
        exp.expectedFulfillmentCount = 2
        let channel = CoChannel<Int>(capacity: 1)
        channel.receiveFuture().whenSuccess {
            XCTAssertEqual($0, 1)
            exp.fulfill()
        }
        channel.sendFuture(.init(result: .success(1)))
        channel.sendFuture(.init(result: .success(2)))
        channel.sendFuture(.init(result: .success(3)))
        channel.receiveFuture().whenSuccess {
            XCTAssertEqual($0, 2)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 1)
    }
    
    func testSender() {
        let exp = expectation(description: "testSender")
        exp.expectedFulfillmentCount = 2
        let sender = CoChannel<Int>().sender
        XCTAssertEqual(sender.bufferType, .unlimited)
        XCTAssertNotNil(try? sender.awaitSend(1))
        sender.sendFuture(.init(result: .success(2)))
        XCTAssertTrue(sender.offer(3))
        XCTAssertEqual(sender.count, 3)
        XCTAssertFalse(sender.isEmpty)
        XCTAssertFalse(sender.isClosed)
        XCTAssertFalse(sender.isCanceled)
        sender.whenComplete { exp.fulfill() }
        sender.whenCanceled { exp.fulfill() }
        sender.cancel()
        XCTAssertFalse(sender.close())
        wait(for: [exp], timeout: 1)
    }
    
    func testMap() {
        let exp = expectation(description: "testMap")
        exp.expectedFulfillmentCount = 2
        let channel = CoChannel<Int>()
        let map2 = channel.map { $0 + 2 }
        channel.offer(7)
        XCTAssertEqual(map2.poll(), 9)
        let (receiver, sender) = channel.pair
        let map = receiver.map { $0 + 1 }.map { $0 + 1 }
        XCTAssertEqual(map.bufferType, sender.bufferType)
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
        map.whenComplete { exp.fulfill() }
        map.whenCanceled { exp.fulfill() }
        map.cancel()
        XCTAssertTrue(map.isCanceled)
        wait(for: [exp], timeout: 5)
    }
    
    
    
    func testCancelFinish() {
        let exp = expectation(description: "testCancelFinished")
        exp.expectedFulfillmentCount = 2
        let channel = CoChannel<Int>()
        channel.whenCanceled { exp.fulfill() }
        channel.cancel()
        channel.whenCanceled { exp.fulfill() }
        wait(for: [exp], timeout: 1)
    }
    
    func testCloseFinish() {
        let exp = expectation(description: "testCancelFinished")
        let channel = CoChannel<Int>()
        channel.whenComplete { exp.fulfill() }
        channel.close()
        wait(for: [exp], timeout: 1)
    }
    
    func testFinish() {
        let exp = expectation(description: "testCancelFinished")
        let channel = _BufferedChannel<Int>(capacity: 1)
        channel.whenFinished { _ in exp.fulfill() }
        channel.finish()
        wait(for: [exp], timeout: 1)
    }
    
    func testCoChannelReceiver() {
        let wrapper = _Channel<Any>()
        _ = try? wrapper.awaitReceive()
        _ = wrapper.receiveFuture()
        _ = wrapper.poll()
        wrapper.whenReceive { _ in }
        wrapper.cancel()
        wrapper.whenComplete {}
        wrapper.whenCanceled {}
        _ = wrapper.count
        _ = wrapper.isEmpty
        _ = wrapper.isClosed
        _ = wrapper.isCanceled
        _ = wrapper.bufferType
        try? wrapper.awaitSend(1)
        wrapper.sendFuture(.init(result: .success(1)))
        _ = wrapper.offer(1)
        _ = wrapper.close()
        _ = wrapper
        _ = CoChannel<Int>.Receiver().whenFinished { _ in }
    }
    
}
