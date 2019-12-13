//
//  SwiftCoroutineTests.swift
//  Tests
//
//  Created by Alex Belozierov on 22.11.19.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import XCTest
import SwiftCoroutine

class SwiftCoroutineTests: XCTestCase {
    
    func testCoroutineSwitch() {
        let expectation = XCTestExpectation(description: "Coroutine switch")
        let date = Date()
        var result = [Int]()
        coroutine {
            result.append(0)
            delay(1)
            XCTAssertDuration(from: date, in: 1..<2)
            result.append(3)
        }
        coroutine {
            result.append(1)
            delay(3)
            XCTAssertDuration(from: date, in: 3..<4)
            XCTAssertEqual(result, (0..<5).map { $0 })
            expectation.fulfill()
        }
        coroutine {
            result.append(2)
            delay(2)
            result.append(4)
            XCTAssertDuration(from: date, in: 2..<3)
        }
        wait(for: [expectation], timeout: 60)
    }
    
    func testCompose() {
        let expectation = XCTestExpectation(description: "Compose execute")
        let expectation2 = XCTestExpectation(description: "Notifiy")
        let session = URLSession.shared
        let future = compose {
            session.data(for: .testImageURL)
            session.data(for: .testImageURL)
            session.data(for: .testImageURL)
        }.transform {
            $0.map { $0.data }
        }
        coroutine {
            defer { expectation.fulfill() }
            guard let dataArray = try? future.await() else { return XCTFail() }
            XCTAssertEqual(dataArray.count, 3)
        }
        future.notify(queue: .global()) {
            XCTAssertEqual(try? $0.get().count, 3)
            expectation2.fulfill()
        }
        wait(for: [expectation, expectation2], timeout: 60)
    }
    
    func testNested() {
        let expectation = XCTestExpectation(description: "Test nested")
        let item1 = DispatchQueue.global().async { () -> Int in
            sleep(2)
            return 5
        }
        let item2 = DispatchQueue.global().async { () -> Int in
            sleep(3)
            return 6
        }
        let date = Date()
        coroutine {
            let current = Coroutine.current
            XCTAssertNotNil(current)
            delay(1)
            XCTAssertDuration(from: date, in: 1..<2)
            delay(1)
            XCTAssertDuration(from: date, in: 2..<3)
            try coroutine {
                delay(2)
                XCTAssertFalse(current === Coroutine.current)
                XCTAssertDuration(from: date, in: 4..<5)
            }.await()
            XCTAssertTrue(current === Coroutine.current)
            XCTAssertDuration(from: date, in: 4..<5)
            delay(1)
            XCTAssertDuration(from: date, in: 5..<6)
            expectation.fulfill()
        }
        coroutine {
            let result = try item1.await() + item2.await()
            XCTAssertEqual(result, 11)
            XCTAssertDuration(from: date, in: 3..<4)
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            coroutine(on: .global()) {
                delay(2)
                XCTAssertDuration(from: date, in: 3..<4)
            }
        }
        wait(for: [expectation], timeout: 60)
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
                DispatchQueue.main.setDispatcher()
                XCTAssertTrue(Thread.isMainThread)
                DispatchQueue.global().setDispatcher()
                XCTAssertFalse(Thread.isMainThread)
            }
        }
        group.notify(queue: .global(), execute: expectation.fulfill)
        wait(for: [expectation], timeout: 60)
    }
    
    func testSyncDispatchCoroutine() {
        let cor1 = Coroutine { $0() }
        let cor2 = Coroutine { $0() }
        var result = [Int]()
        cor1.start {
            XCTAssertTrue(Coroutine.current === cor1)
            result.append(0)
            cor1.suspend()
            XCTAssertTrue(Coroutine.current === cor1)
            result.append(3)
        }
        XCTAssertNil(Coroutine.current)
        result.append(1)
        cor2.start {
            XCTAssertTrue(Coroutine.current === cor2)
            result.append(2)
            cor1.resume()
            XCTAssertTrue(Coroutine.current === cor2)
            result.append(4)
        }
        XCTAssertNil(Coroutine.current)
        XCTAssertEqual(result, (0..<5).map { $0 })
    }
    
    func testLazyPromise() {
        let expectation = XCTestExpectation(description: "Lazy promise test")
        let item1 = CoLazyPromise<Int> {
            sleep(2)
            $0(.success(5))
        }
        let item2 = CoLazyPromise<Int>(queue: .global()) {
            sleep(1)
            $0(.success(6))
        }
        let date = Date()
        coroutine(on: .global()) {
            let result = try item1.await() + item2.await()
            XCTAssertEqual(result, 11)
            XCTAssertDuration(from: date, in: 3..<4)
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 10)
    }
    
}


