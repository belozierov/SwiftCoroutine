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
        let session = URLSession.shared
        let future = compose {
            session.data(for: .testImageURL)
            session.data(for: .testImageURL)
            session.data(for: .testImageURL)
        }.transform {
            $0.map { $0.data }.compactMap(UIImage.init)
        }
        coroutine {
            defer { expectation.fulfill() }
            guard let images = try? future.await() else { return XCTFail() }
            XCTAssertEqual(images.count, 3)
        }
        wait(for: [expectation], timeout: 60)
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
            delay(1)
            XCTAssertDuration(from: date, in: 1..<2)
            delay(1)
            XCTAssertDuration(from: date, in: 2..<3)
            try coroutine {
                delay(2)
                XCTAssertDuration(from: date, in: 4..<5)
            }.await()
            XCTAssertDuration(from: date, in: 4..<5)
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
        coroutine {
            XCTAssertTrue(Thread.isMainThread)
            DispatchQueue.global().switchTo()
            XCTAssertFalse(Thread.isMainThread)
            DispatchQueue.main.switchTo()
            XCTAssertTrue(Thread.isMainThread)
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 60)
    }
    
}


