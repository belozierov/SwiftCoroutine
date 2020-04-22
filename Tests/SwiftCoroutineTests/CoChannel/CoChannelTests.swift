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
            for value in channel {
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
    
    func testInsideCoroutine() {
        let exp = expectation(description: "testInsideCoroutine")
        let channel = CoChannel<Int>(maxBufferSize: 1)
        var set = Set<Int>()
        DispatchQueue.global().startCoroutine {
            for value in channel {
                set.insert(value)
            }
            XCTAssertNil(try? channel.awaitReceive())
            XCTAssertEqual(set.count, 10_000)
            exp.fulfill()
        }
        DispatchQueue.global().startCoroutine {
            for index in (0..<10_000) {
                try channel.awaitSend(index)
            }
            channel.close()
            try channel.awaitSend(-1)
        }
        wait(for: [exp], timeout: 20)
    }
    
    func testOffer() {
        let channel = CoChannel<Int>(maxBufferSize: 1)
        ImmediateScheduler().startCoroutine {
            XCTAssertEqual(try? channel.awaitReceive(), 1)
        }
        channel.offer(1)
        channel.offer(2)
        channel.offer(3)
        XCTAssertEqual(channel.count, 1)
        channel.whenReceive {
            XCTAssertEqual(try? $0.get(), 2)
        }
        channel.offer(4)
        XCTAssertEqual(channel.poll(), 4)
        XCTAssertTrue(channel.isEmpty)
        channel.offer(5)
        XCTAssertFalse(channel.isEmpty)
        XCTAssertEqual(channel.makeIterator().next(), 5)
        XCTAssertNil(channel.makeIterator().next())
        channel.whenReceive {
            XCTAssertEqual(try? $0.get(), 6)
        }
        channel.offer(6)
    }
    
    func testFuture() {
        let channel = CoChannel<Int>(maxBufferSize: 1)
        channel.receiveFuture().whenSuccess {
            XCTAssertEqual($0, 1)
        }
        channel.sendFuture(.init(result: .success(1)))
        channel.sendFuture(.init(result: .success(2)))
        channel.sendFuture(.init(result: .success(3)))
        channel.receiveFuture().whenSuccess {
            XCTAssertEqual($0, 2)
        }
    }
    
    func testCancel() {
        let channel = CoChannel<Int>(maxBufferSize: 0)
        ImmediateScheduler().startCoroutine {
            XCTAssertThrowError(CoChannelError.canceled) { try channel.awaitSend(0) }
        }
        XCTAssertFalse(channel.isCanceled)
        channel.cancel()
        XCTAssertThrowError(CoChannelError.canceled) { try channel.awaitReceive() }
        XCTAssertThrowError(CoChannelError.canceled) { try channel.awaitSend(1) }
        XCTAssertFalse(channel.offer(1))
        channel.whenReceive { result in
            XCTAssertThrowError(CoChannelError.canceled) { try result.get() }
        }
        XCTAssertNil(channel.poll())
        XCTAssertTrue(channel.isCanceled)
    }

    func testCancel2() {
        let channel = CoChannel<Int>(maxBufferSize: 0)
        ImmediateScheduler().startCoroutine {
            XCTAssertThrowError(CoChannelError.canceled) { try channel.awaitReceive() }
        }
        channel.cancel()
    }
    
    func testClose() {
        let channel = CoChannel<Int>(maxBufferSize: 1)
        channel.whenReceive { result in
            XCTAssertThrowError(CoChannelError.closed) { try result.get() }
        }
        XCTAssertFalse(channel.isClosed)
        XCTAssertTrue(channel.close())
        XCTAssertFalse(channel.close())
        XCTAssertTrue(channel.isClosed)
        channel.whenReceive { result in
            XCTAssertThrowError(CoChannelError.closed) { try result.get() }
        }
        channel.receiveFuture().whenFailure {
            if let error = $0 as? CoChannelError {
                XCTAssertEqual(error, CoChannelError.closed)
            } else {
                XCTFail()
            }
        }
        channel.sendFuture(.init(result: .success(1)))
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
    
}
