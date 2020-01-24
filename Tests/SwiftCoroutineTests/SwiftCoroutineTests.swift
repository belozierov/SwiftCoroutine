//
//  SwiftCoroutineTests.swift
//  Tests
//
//  Created by Alex Belozierov on 22.11.19.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import XCTest
import SwiftCoroutine

#if !os(macOS)
import FoundationNetworking
import Foundation
#endif

class SwiftCoroutineTests: XCTestCase {
    
    func testCoroutineSwitch() {
        let expectation = XCTestExpectation(description: "Coroutine switch")
        let date = Date()
        var result = [Int]()
        coroutine {
            result.append(0)
            try Coroutine.delay(1)
            XCTAssertDuration(from: date, in: 1..<2)
            result.append(3)
        }
        coroutine {
            result.append(1)
            try Coroutine.delay(3)
            XCTAssertDuration(from: date, in: 3..<4)
            XCTAssertEqual(result, (0..<5).map { $0 })
            expectation.fulfill()
        }
        coroutine {
            result.append(2)
            try Coroutine.delay(2)
            result.append(4)
            XCTAssertDuration(from: date, in: 2..<3)
        }
        wait(for: [expectation], timeout: 5)
    }
    
    func testCompose() {
        let expectation = XCTestExpectation(description: "Compose execute")
        let expectation2 = XCTestExpectation(description: "Notifiy")
        let session = URLSession.shared
        let future = compose { () -> [CoFuture<URLSession.DataResponse>] in
            session.dataTaskFuture(for: .testImageURL)
            session.dataTaskFuture(for: .testImageURL)
            session.dataTaskFuture(for: .testImageURL)
        }.transformOutput {
            $0.map { $0.data }
        }
        coroutine {
            defer { expectation.fulfill() }
            guard let dataArray = try? future.await() else { return XCTFail() }
            XCTAssertEqual(dataArray.count, 3)
        }
        future.onSuccess(on: .global) {
            XCTAssertEqual($0.count, 3)
            expectation2.fulfill()
        }
        wait(for: [expectation, expectation2], timeout: 60)
    }
    
    func testNested() {
        let expectation = XCTestExpectation(description: "Test nested")
        let item1 = async { () -> Int in
            sleep(2)
            return Int(5)
        }
        let item2 = async { () -> Int in
            sleep(3)
            return 6
        }
        let date = Date()
        coroutine {
            let current = try Coroutine.current()
            XCTAssertNotNil(current)
            try Coroutine.delay(1)
            XCTAssertDuration(from: date, in: 1..<2)
            subroutine {
                let item3 = async { () -> Int in
                    sleep(1)
                    return 6
                }
                XCTAssertEqual(try? item3.await(), 6)
                XCTAssertTrue(current.isCurrent)
            }
            XCTAssertDuration(from: date, in: 2..<3)
            let future: CoFuture<Void> = coroutine {
                try Coroutine.delay(2)
                XCTAssertFalse(current.isCurrent)
                XCTAssertDuration(from: date, in: 4..<5)
            }
            try future.await()
            XCTAssertTrue(current.isCurrent)
            XCTAssertDuration(from: date, in: 4..<5)
            try Coroutine.delay(1)
            XCTAssertDuration(from: date, in: 5..<6)
            expectation.fulfill()
        }
        coroutine {
            let result = try item1.await() + item2.await()
            XCTAssertEqual(result, 11)
            XCTAssertDuration(from: date, in: 3..<4)
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            coroutine(on: .global) {
                try Coroutine.delay(2)
                XCTAssertDuration(from: date, in: 3..<4)
            }
        }
        wait(for: [expectation], timeout: 10)
    }
    
    func testGenerator() {
        var items = [2, 3, 1, 4, 2, 4]
        let generator = Generator<((Int, Int) -> Bool) -> Void> { yield in
            items.sort { left, right in
                var result = false
                yield { result = $0(left, right) }
                return result
            }
        }
        while let next = generator.next() { next(>) }
        XCTAssertEqual(items, items.sorted(by: >))
    }
    
    func testDispatchSwitch() {
        let expectation = XCTestExpectation(description: "Dispatch switch")
        let group = DispatchGroup()
        for _ in 0..<10_000 {
            DispatchQueue.global().coroutine(group: group) {
                XCTAssertFalse(Thread.isMainThread)
                try Coroutine.setDispatcher(.main)
                XCTAssertTrue(Thread.isMainThread)
                try Coroutine.setDispatcher(.global)
                XCTAssertFalse(Thread.isMainThread)
            }
        }

        #if os(macOS)
        group.notify(queue: .global(), execute: expectation.fulfill)
        #else
        group.notify(queue: .global(), work: expectation.fulfill)
        #endif

        wait(for: [expectation], timeout: 60)
    }
    
    func testSyncDispatchCoroutine() {
        let cor1 = Coroutine()
        let cor2 = Coroutine()
        var result = [Int]()
        cor1.start {
            XCTAssertTrue(cor1.isCurrent)
            result.append(0)
            cor1.suspend()
            XCTAssertTrue(cor1.isCurrent)
            result.append(3)
        }
        XCTAssertNil(try? Coroutine.current())
        result.append(1)
        cor2.start {
            XCTAssertTrue(cor2.isCurrent)
            result.append(2)
            cor1.resume()
            XCTAssertTrue(cor2.isCurrent)
            result.append(4)
        }
        XCTAssertNil(try? Coroutine.current())
        XCTAssertEqual(result, (0..<5).map { $0 })
    }
    
    func testAwaitTimeOut() {
        let expectation = XCTestExpectation(description: "Await timeout")
        let date = Date()
        let future = async(on: .global) { () -> Int in
            sleep(2)
            return 5
        }
        coroutine {
            _ = try future.await(timeout: .now() + 1)
        }.onFutureError(.timeout) {
            XCTAssertDuration(from: date, in: 1..<2)
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 5)
    }
    
    func testFutureWait() {
        let result = try? async { () -> Int in
            sleep(1)
            return 1
        }.wait()
        XCTAssertEqual(result, 1)
    }
    
    func testPoolPerformence() {
        let withPool = performanceTest { Coroutine.newFromPool(dispatcher: $0) }
        let withoutPool = performanceTest { Coroutine(dispatcher: $0) }
        let percent = withoutPool / withPool - 1
        print("withPool faster for \(Int(percent * 100))%")
        XCTAssert(percent > 0)
    }
    
    private func performanceTest(creator: (Coroutine.Dispatcher) -> Coroutine) -> Double {
        let group = DispatchGroup()
        let dispatcher = Coroutine.Dispatcher.dispatchQueue(.global(), group: group)
        let date = Date()
        for _ in 0..<10_000 { creator(dispatcher).start {} }
        group.wait()
        return Date().timeIntervalSince(date)
    }
    
    func testCoroutineState() {
        let cor = Coroutine()
        XCTAssertEqual(cor.state, .prepared)
        cor.start {
            XCTAssertEqual(cor.state, .running)
            cor.suspend {
                XCTAssertEqual(cor.state, .suspended)
                cor.resume()
            }
            XCTAssertEqual(cor.state, .running)
        }
        XCTAssertEqual(cor.state, .prepared)
    }
    
}
